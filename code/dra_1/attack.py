import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.models import Decoder
from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise

def visualize_reconstruction(originals, reconstructions, task_names, save_path):
    """
    Generates and saves plots comparing original and reconstructed signals for multiple tasks.
    """
    num_samples_to_plot = len(next(iter(originals.values())))
    num_tasks = len(task_names)
    
    fig, axes = plt.subplots(num_samples_to_plot * num_tasks, 3, 
                             figsize=(18, 3 * num_samples_to_plot * num_tasks), 
                             squeeze=False)
    fig.suptitle("Data Reconstruction Results for All Tasks", fontsize=20)

    for i in range(num_samples_to_plot):
        for j, task in enumerate(task_names):
            row_idx = i * num_tasks + j
            
            original_np = originals[task][i].squeeze(0).numpy()
            recon_np = reconstructions[task][i].squeeze(0).numpy()

            # Plot Real Part
            axes[row_idx, 0].plot(original_np[0][:100], label='Original')
            axes[row_idx, 0].plot(recon_np[0][:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 0].set_ylabel(f"Sample {i+1}\n{task.upper()}")
            axes[row_idx, 0].set_title("Real Part")
            axes[row_idx, 0].legend()

            # Plot Imaginary Part
            axes[row_idx, 1].plot(original_np[1][:100], label='Original')
            axes[row_idx, 1].plot(recon_np[1][:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 1].set_title("Imaginary Part")
            axes[row_idx, 1].legend()

            # Plot Magnitude
            original_mag = np.sqrt(original_np[0]**2 + original_np[1]**2)
            recon_mag = np.sqrt(recon_np[0]**2 + recon_np[1]**2)
            axes[row_idx, 2].plot(original_mag[:100], label='Original')
            axes[row_idx, 2].plot(recon_mag[:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 2].set_title("Magnitude")
            axes[row_idx, 2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved reconstruction visualization to {save_path}")


def plot_epoch_samples(decoder, sample_batch, epoch, save_dir, device, task_names=['rf', 'cfo', 'channel']):
    """
    Plot sample reconstructions at each epoch to monitor training progress.
    """
    decoder.eval()
    with torch.no_grad():
        rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation_batch, filenames = sample_batch
        
        # Process data
        activation_batch = activation_batch[:2].squeeze(1).to(device)  # Use first 2 samples
        true_rf = rf_x[:2].squeeze(1).to(device)
        true_cfo = cfo_x[:2].squeeze(1).to(device)
        true_channel = channel_x[:2].squeeze(1).to(device)
        
        # Get reconstructions
        reconstructed_x = decoder(activation_batch)
        
        # DEBUG: Print actual values being plotted
        print(f"\n  🔍 DEBUG - Epoch {epoch} reconstruction values:")
        for task in ['rf', 'cfo', 'channel']:
            pred_data = reconstructed_x[task]
            print(f"    {task.upper()} pred - Mean: {pred_data.mean().item():.6f}, Std: {pred_data.std().item():.6f}, "
                  f"Min: {pred_data.min().item():.6f}, Max: {pred_data.max().item():.6f}")
            print(f"    {task.upper()} pred sample[0] first 5 values: {pred_data[0, 0, :5].cpu().numpy()}")
        
        # Create plot
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))  # 2 samples x 6 plots (3 tasks x 2 components each)
        fig.suptitle(f'Epoch {epoch} - Sample Reconstructions', fontsize=16)
        
        samples_data = [
            (true_rf, reconstructed_x['rf'], 'RF', 0),
            (true_cfo, reconstructed_x['cfo'], 'CFO', 2), 
            (true_channel, reconstructed_x['channel'], 'Channel', 4)
        ]
        
        for sample_idx in range(2):  # First 2 samples
            for true_data, pred_data, task_name, col_offset in samples_data:
                true_np = true_data[sample_idx].cpu().numpy()
                pred_np = pred_data[sample_idx].cpu().numpy()
                
                # Real part
                axes[sample_idx, col_offset].plot(true_np[0][:100], 'b-', label='True', alpha=0.8)
                axes[sample_idx, col_offset].plot(pred_np[0][:100], 'r--', label='Pred', alpha=0.8)
                axes[sample_idx, col_offset].set_title(f'{task_name} Real (Sample {sample_idx+1})')
                axes[sample_idx, col_offset].legend()
                axes[sample_idx, col_offset].grid(True, alpha=0.3)
                
                # Imaginary part
                axes[sample_idx, col_offset+1].plot(true_np[1][:100], 'b-', label='True', alpha=0.8)
                axes[sample_idx, col_offset+1].plot(pred_np[1][:100], 'r--', label='Pred', alpha=0.8)
                axes[sample_idx, col_offset+1].set_title(f'{task_name} Imag (Sample {sample_idx+1})')
                axes[sample_idx, col_offset+1].legend()
                axes[sample_idx, col_offset+1].grid(True, alpha=0.3)
                
                # Calculate and display correlation
                corr_real = np.corrcoef(true_np[0], pred_np[0])[0, 1]
                corr_imag = np.corrcoef(true_np[1], pred_np[1])[0, 1]
                mse = np.mean((true_np - pred_np)**2)
                
                # Add text with metrics
                metrics_text = f'Corr: {corr_real:.3f}/{corr_imag:.3f}\nMSE: {mse:.3f}'
                axes[sample_idx, col_offset].text(0.02, 0.98, metrics_text, 
                                                 transform=axes[sample_idx, col_offset].transAxes,
                                                 verticalalignment='top', fontsize=8,
                                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        epoch_plot_path = os.path.join(save_dir, f'epoch_{epoch:03d}_samples.png')
        plt.savefig(epoch_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Also save compact stats
        stats = {}
        for true_data, pred_data, task_name, _ in samples_data:
            true_np = true_data.cpu().numpy()
            pred_np = pred_data.cpu().numpy()
            
            # Average correlation across samples and components
            corrs = []
            for i in range(min(2, true_np.shape[0])):
                corr_real = np.corrcoef(true_np[i, 0], pred_np[i, 0])[0, 1]
                corr_imag = np.corrcoef(true_np[i, 1], pred_np[i, 1])[0, 1]
                corrs.extend([corr_real, corr_imag])
            
            avg_corr = np.mean([c for c in corrs if not np.isnan(c)])
            mse = np.mean((true_np - pred_np)**2)
            
            stats[task_name.lower()] = {
                'correlation': avg_corr,
                'mse': mse,
                'pred_mean': pred_np.mean(),
                'pred_std': pred_np.std()
            }
        
        print(f"  Epoch {epoch} sample stats:")
        for task, stat in stats.items():
            print(f"    {task.upper()}: Corr={stat['correlation']:.4f}, MSE={stat['mse']:.4f}, "
                  f"Pred(μ={stat['pred_mean']:.3f}, σ={stat['pred_std']:.3f})")
        
        return stats


def plot_epoch_trends(epoch_stats, save_dir):
    """
    Plot trends of reconstruction quality across epochs.
    """
    if not epoch_stats:
        return
    
    epochs = list(range(1, len(epoch_stats) + 1))
    
    # Extract metrics for each task
    tasks = ['rf', 'cfo', 'channel']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Progress - Reconstruction Quality Trends', fontsize=16)
    
    for task_idx, task in enumerate(tasks):
        correlations = [stats[task]['correlation'] for stats in epoch_stats]
        mses = [stats[task]['mse'] for stats in epoch_stats]
        pred_means = [stats[task]['pred_mean'] for stats in epoch_stats]
        pred_stds = [stats[task]['pred_std'] for stats in epoch_stats]
        
        # Correlation plot
        axes[0, task_idx].plot(epochs, correlations, 'o-', linewidth=2, markersize=4)
        axes[0, task_idx].set_title(f'{task.upper()} Correlation')
        axes[0, task_idx].set_xlabel('Epoch')
        axes[0, task_idx].set_ylabel('Correlation')
        axes[0, task_idx].grid(True, alpha=0.3)
        axes[0, task_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # MSE plot
        axes[1, task_idx].plot(epochs, mses, 'o-', linewidth=2, markersize=4, color='red')
        axes[1, task_idx].set_title(f'{task.upper()} MSE')
        axes[1, task_idx].set_xlabel('Epoch')
        axes[1, task_idx].set_ylabel('MSE')
        axes[1, task_idx].grid(True, alpha=0.3)
        axes[1, task_idx].set_yscale('log')
        
        # Add final values as text
        final_corr = correlations[-1] if correlations else 0
        final_mse = mses[-1] if mses else 0
        axes[0, task_idx].text(0.05, 0.95, f'Final: {final_corr:.3f}', 
                              transform=axes[0, task_idx].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, task_idx].text(0.05, 0.95, f'Final: {final_mse:.3f}', 
                              transform=axes[1, task_idx].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    trend_plot_path = os.path.join(save_dir, 'epoch_trends.png')
    plt.savefig(trend_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved epoch trends to {trend_plot_path}")
    
    # Also save the stats as JSON for further analysis
    import json
    stats_path = os.path.join(save_dir, 'epoch_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(epoch_stats, f, indent=2)
    print(f"Saved epoch statistics to {stats_path}")


def train_decoder(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths based on experiment_path
    activation_dir = os.path.join(cli_args.experiment_path, 'activations')
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"'activations' directory not found in {cli_args.experiment_path}. Expected it at: {activation_dir}")
    print(f"Found activations directory: {activation_dir}")

    # Use PKL_FILE_PATH environment variable for partition file
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if pkl_file_path is None:
        raise ValueError("PKL_FILE_PATH environment variable not set. Please set it to the directory containing the partition file.")
    
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file not found at: {partition_file}")
    print(f"Using partition file: {partition_file}")

    # Create save directory
    save_dir = os.path.join(cli_args.experiment_path, 'attack_results', cli_args.noise_type, f"level_{str(cli_args.noise_level).replace('.', '_')}")
    os.makedirs(save_dir, exist_ok=True)
    cli_args.save_path = os.path.join(save_dir, 'best_adversary.pt')
    print(f"Saving results to: {save_dir}")

    # Save attack arguments for reproducibility
    attack_args = {
        'experiment_path': cli_args.experiment_path,
        'noise_type': cli_args.noise_type,
        'noise_level': cli_args.noise_level,
        'epochs': cli_args.epochs,
        'lr': cli_args.lr,
        'batch_size': cli_args.batch_size,
        'patience': cli_args.patience,
        'rf_loss_weight': cli_args.rf_loss_weight,
        'latent_dim': cli_args.latent_dim,
        'fim_samples': cli_args.fim_samples if cli_args.noise_type == 'nonisotropic' else None
    }
    
    with open(os.path.join(save_dir, 'attack_args.json'), 'w') as f:
        json.dump(attack_args, f, indent=4)

    # Load the dataset containing activations
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions.get('train')
    val_files = partitions.get('val')
    test_files = partitions.get('test')

    if not train_files or not val_files or not test_files:
        raise ValueError("Could not find 'train', 'val', and 'test' keys in the partition file.")

    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=train_files)
    val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=cli_args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)

    # Initialize the Decoder model and optimizer
    decoder = Decoder(latent_dim=cli_args.latent_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=cli_args.lr)
    criterion = nn.L1Loss()

    # Setup for nonisotropic noise if needed
    L, V = None, None
    if cli_args.noise_type == 'nonisotropic':
        print("\n===== Calculating FIM for Nonisotropic Noise =====")
        # For nonisotropic noise, we need to compute FIM using training data
        # This is a simplified version - in practice you'd want to load task heads
        # and compute FIM properly like in check_utility.py
        print("Warning: Nonisotropic noise requires FIM calculation with task heads.")
        print("For now, falling back to isotropic noise. Please implement FIM calculation.")
        cli_args.noise_type = 'isotropic'

    # --- DEBUG: Focus only on RF reconstruction ---
    TASKS = ['rf', 'cfo', 'channel']

    best_val_loss = float('inf')
    patience_counter = 0
    
    # Get a fixed sample batch for monitoring across epochs
    print("Getting sample batch for epoch monitoring...")
    sample_batch = None
    for batch in val_loader:
        sample_batch = batch
        break
    
    # Store epoch statistics for plotting trends
    epoch_stats = []

    print("Starting decoder training...")
    for epoch in range(cli_args.epochs):
        # --- Training Phase ---
        decoder.train()
        train_losses = {task: 0.0 for task in TASKS}
        
        # --- Check weight at start of epoch ---
        weight_at_epoch_start = decoder.shared_decoder[0].weight.data[0, 0].item()
        
        for i, (rf_x, _, cfo_x, _, channel_x, _, activation_batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cli_args.epochs} [Train]")):
            activation_batch = activation_batch.squeeze(1).to(device)
            
            # --- Input data stats (for the first batch of each epoch) ---
            if i == 0:
                print(f"\n--- Data stats (Epoch {epoch+1}, first batch) ---")
                # Stats for the latent activation input to the decoder
                print(f"  Latent Activation -> Mean: {activation_batch.mean().item():.3e}, Std: {activation_batch.std().item():.3e}, Min: {activation_batch.min().item():.3e}, Max: {activation_batch.max().item():.3e}")

                # Stats for the ground truth signals
                true_rf = rf_x.squeeze(1).to(device)
                true_cfo = cfo_x.squeeze(1).to(device)
                true_channel = channel_x.squeeze(1).to(device)

                print(f"  RF Ground Truth     -> Mean: {true_rf.mean().item():.3e}, Std: {true_rf.std().item():.3e}, Min: {true_rf.min().item():.3e}, Max: {true_rf.max().item():.3e}")
                print(f"  CFO Ground Truth    -> Mean: {true_cfo.mean().item():.3e}, Std: {true_cfo.std().item():.3e}, Min: {true_cfo.min().item():.3e}, Max: {true_cfo.max().item():.3e}")
                print(f"  Channel Ground Truth-> Mean: {true_channel.mean().item():.3e}, Std: {true_channel.std().item():.3e}, Min: {true_channel.min().item():.3e}, Max: {true_channel.max().item():.3e}")
                print("--- End data stats ---\n")

            # Apply noise based on type
            if cli_args.noise_type == 'isotropic':
                activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
            elif cli_args.noise_type == 'nonisotropic':
                activation_batch = inject_nonisotropic_noise(activation_batch, cli_args.noise_level, L, V)
            # For 'none', no noise is applied
            
            true_rf = rf_x.squeeze(1).to(device)
            true_cfo = cfo_x.squeeze(1).to(device)
            true_channel = channel_x.squeeze(1).to(device)
            ground_truth = {'rf': true_rf, 'cfo': true_cfo, 'channel': true_channel}

            
            reconstructed_x = decoder(activation_batch)

            #print(reconstructed_x['rf'] - true_rf_input)

            
            optimizer.zero_grad()
            
            # Only calculate loss for the RF task
            loss_rf = criterion(reconstructed_x['rf'], ground_truth['rf'])
            loss_cfo = criterion(reconstructed_x['cfo'], ground_truth['cfo'])
            loss_channel = criterion(reconstructed_x['channel'], ground_truth['channel'])
            
            total_loss = (cli_args.rf_loss_weight * loss_rf) + loss_cfo + loss_channel
            
            total_loss.backward()
            optimizer.step()
            
            train_losses['rf'] += loss_rf.item()
            train_losses['cfo'] += loss_cfo.item()
            train_losses['channel'] += loss_channel.item()

        avg_train_losses = {task: loss / len(train_loader) for task, loss in train_losses.items()}

        # --- Check weight at end of epoch and report change ---
        weight_at_epoch_end = decoder.shared_decoder[0].weight.data[0, 0].item()
        print(f"\n--- Epoch {epoch+1} Weight Change Summary ---")
        print(f"  Weight at START of epoch: {weight_at_epoch_start:.6f}")
        print(f"  Weight at END of epoch:   {weight_at_epoch_end:.6f}")
        print(f"  Total change this epoch:  {weight_at_epoch_end - weight_at_epoch_start:.6e}")
        print("--- End Summary ---\n")

        # --- Validation Phase ---
        decoder.eval()
        val_losses = {task: 0.0 for task in TASKS}
        with torch.no_grad():
            for rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cli_args.epochs} [Val]"):
                activation_batch = activation_batch.squeeze(1).to(device)
                
                # Apply noise based on type
                if cli_args.noise_type == 'isotropic':
                    activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
                elif cli_args.noise_type == 'nonisotropic':
                    activation_batch = inject_nonisotropic_noise(activation_batch, cli_args.noise_level, L, V)
                
                true_rf = rf_x.squeeze(1).to(device)
                true_cfo = cfo_x.squeeze(1).to(device)
                true_channel = channel_x.squeeze(1).to(device)
                ground_truth = {'rf': true_rf, 'cfo': true_cfo, 'channel': true_channel}

                reconstructed_x = decoder(activation_batch)
                
                for task in TASKS:
                    val_losses[task] += criterion(reconstructed_x[task], ground_truth[task]).item()

        avg_val_losses = {task: loss / len(val_loader) for task, loss in val_losses.items()}
        avg_val_loss = (cli_args.rf_loss_weight * avg_val_losses['rf']) + avg_val_losses['cfo'] + avg_val_losses['channel']


        # --- Logging ---
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train L1 Loss -> " + " | ".join([f"{task.upper()}: {avg_train_losses[task]:.3e}" for task in TASKS]))
        print(f"  Val L1 Loss   -> " + " | ".join([f"{task.upper()}: {avg_val_losses[task]:.3e}" for task in TASKS]))


        # --- Checkpointing and Early Stopping (based on raw total validation loss) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), cli_args.save_path)
            print(f"  (Raw Val Loss: {avg_val_loss:.6f}) -> New best model saved.")
        else:
            patience_counter += 1
            print(f"  (Raw Val Loss: {avg_val_loss:.6f}) -> No improvement. Patience: {patience_counter}/{cli_args.patience}")

        # --- Plot sample outputs for this epoch ---
        if sample_batch is not None:
            epoch_sample_stats = plot_epoch_samples(decoder, sample_batch, epoch + 1, save_dir, device)
            epoch_stats.append(epoch_sample_stats)
        
        if patience_counter >= cli_args.patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished. Evaluating best model on the test set...")
    
    # Load the best performing model
    decoder.load_state_dict(torch.load(cli_args.save_path))
    decoder.to(device)
    decoder.eval()
    
    test_losses = {task: 0.0 for task in TASKS}
    original_samples = {task: [] for task in TASKS}
    reconstructed_samples = {task: [] for task in TASKS}
    
    with torch.no_grad():
        for rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ in tqdm(test_loader, desc="Testing"):
            activation_batch = activation_batch.squeeze(1).to(device)
            
            # Apply noise based on type
            if cli_args.noise_type == 'isotropic':
                activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
            elif cli_args.noise_type == 'nonisotropic':
                activation_batch = inject_nonisotropic_noise(activation_batch, cli_args.noise_level, L, V)
            
            true_rf = rf_x.squeeze(1).to(device)
            true_cfo = cfo_x.squeeze(1).to(device)
            true_channel = channel_x.squeeze(1).to(device)
            ground_truth = {'rf': true_rf, 'cfo': true_cfo, 'channel': true_channel}

            reconstructed_x = decoder(activation_batch)
            
            for task in TASKS:
                test_losses[task] += criterion(reconstructed_x[task], ground_truth[task]).item()


            # Save the first batch for visualization
            if not original_samples['rf']:
                num_to_visualize = min(5, activation_batch.size(0))
                for task in TASKS:
                    original_samples[task].extend(list(torch.split(ground_truth[task].cpu(), 1))[:num_to_visualize])
                    reconstructed_samples[task].extend(list(torch.split(reconstructed_x[task].cpu(), 1))[:num_to_visualize])
    
    avg_test_losses = {task: loss / len(test_loader) for task, loss in test_losses.items()}

    print("\n--- Final Test Results ---")
    print("  --- Per-Task Breakdown ---")
    for task in TASKS:
        print(f"  {task.upper()} L1 Loss: {avg_test_losses[task]:.3e} (Raw Loss: {avg_test_losses[task]:.6f})")

    # Visualize results
    plot_path = os.path.join(save_dir, 'reconstruction_results_all_tasks.png')
    print(f"Saving reconstruction visualization to {plot_path}")
    visualize_reconstruction(original_samples, reconstructed_samples, TASKS, plot_path)
    
    # Plot epoch trends
    print(f"\nGenerating epoch trend analysis...")
    plot_epoch_trends(epoch_stats, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Decoder to reconstruct data from latent activations.")
    
    # Required paths
    parser.add_argument('--experiment_path', type=str, required=True, help="Path to the experiment results directory, containing the model checkpoint and an 'activations' subdirectory.")
    
    # Noise configuration
    parser.add_argument('--noise_type', type=str, default='none', choices=['isotropic', 'nonisotropic', 'none'], help="Type of noise to inject during training.")
    parser.add_argument('--noise_level', type=float, default=0.0, help="Standard deviation of noise to add to activations.")
    parser.add_argument('--fim_samples', type=int, default=1000, help="Number of samples to use for FIM calculation for nonisotropic noise.")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the decoder training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")
    parser.add_argument('--rf_loss_weight', type=float, default=4, help="Weight to apply to the RF reconstruction loss.")
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help="Dimension of the latent space (after flattening).")

    cli_args = parser.parse_args()
    train_decoder(cli_args) 