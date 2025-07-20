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
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.models import Decoder
from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise

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
    print(f"Saved reconstruction visualization to {save_path}")


def train_decoder(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset containing activations
    with open(cli_args.partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions.get('train')
    val_files = partitions.get('val')
    test_files = partitions.get('test')

    if not train_files or not val_files or not test_files:
        raise ValueError("Could not find 'train', 'val', and 'test' keys in the partition file.")

    train_dataset = ActivationDataset(activation_dir=cli_args.activation_dir, file_list=train_files)
    val_dataset = ActivationDataset(activation_dir=cli_args.activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=cli_args.activation_dir, file_list=test_files, test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=cli_args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)

    # Initialize the Decoder model and optimizer
    latent_dim = 512  # This should match the flattened output of the encoder (e.g., 2 * 256)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=cli_args.lr)
    criterion = nn.L1Loss()

    # --- DEBUG: Focus only on RF reconstruction ---
    TASKS = ['rf', 'cfo', 'channel']

    best_val_loss = float('inf')
    patience_counter = 0

    # Create save directory if it doesn't exist
    save_dir = os.path.dirname(cli_args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Starting decoder training...")
    for epoch in range(cli_args.epochs):
        # --- Training Phase ---
        decoder.train()
        train_losses = {task: 0.0 for task in TASKS}
        
        # --- Check weight at start of epoch ---
        weight_at_epoch_start = decoder.rf_decoder[0].weight.data[0, 0].item()
        
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

            #activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
            
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
        weight_at_epoch_end = decoder.rf_decoder[0].weight.data[0, 0].item()
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
                activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
                
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
            activation_batch = inject_isotropic_noise(activation_batch, cli_args.noise_level)
            
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
    visualize_reconstruction(original_samples, reconstructed_samples, TASKS, 'reconstruction_results_all_tasks.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Decoder to reconstruct data from latent activations.")
    
    # Required paths
    parser.add_argument('--activation_dir', type=str, required=True, help="Directory containing pre-computed activation files.")
    parser.add_argument('--partition_file', type=str, required=True, help="Path to the data partition file.")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the decoder training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")
    parser.add_argument('--noise_level', type=float, default=0.0, help="Standard deviation of isotropic Gaussian noise to add to activations.")
    parser.add_argument('--rf_loss_weight', type=float, default=4, help="Weight to apply to the RF reconstruction loss.")
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help="Dimension of the latent space (after flattening).")
    
    # Output options
    parser.add_argument('--save_path', type=str, default='results_attacks/trained_decoder.pt', help="Path to save the trained decoder model.")

    cli_args = parser.parse_args()
    train_decoder(cli_args) 