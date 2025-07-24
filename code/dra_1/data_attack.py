import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

# Add parent directories to path to import modules from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rep_lr')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.py_datasets import ActivationDataset
from dra_1.model_loader import load_model_for_attack

def visualize_reconstruction(originals, reconstructions, task_names, save_path):
    """
    Generates and saves plots comparing original and reconstructed signals for multiple tasks.
    """
    num_samples_to_plot = len(next(iter(originals.values())))
    num_tasks = len(task_names)
    
    fig, axes = plt.subplots(num_samples_to_plot * num_tasks, 3, 
                             figsize=(18, 3 * num_samples_to_plot * num_tasks), 
                             squeeze=False)
    fig.suptitle("Data Reconstruction Attack Results", fontsize=20)

    for i in range(num_samples_to_plot):
        for j, task in enumerate(task_names):
            row_idx = i * num_tasks + j
            
            # Squeeze out all singleton dimensions to get (2, length) for plotting
            original_np = originals[task][i].squeeze().numpy()
            recon_np = reconstructions[task][i].squeeze().numpy()

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

def reconstruct_data(cli_args):
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model and Training Args ---
    # Determine experiment directory from the model path
    if os.path.isdir(cli_args.model_path):
        experiment_dir = cli_args.model_path
    else:
        experiment_dir = os.path.dirname(cli_args.model_path)

    # Load model using the new utility
    model_data = load_model_for_attack(experiment_dir, device)
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded and frozen successfully. MTL: {is_mtl}")
    
    # Support both MTL and single-task models
    if is_mtl:
        TASKS = train_args.task
    else:
        TASKS = [train_args.task]

    # --- Load Data ---
    with open(cli_args.partition_file, 'rb') as f:
        partitions = pickle.load(f)
    test_files = partitions.get('test')
    if not test_files:
        raise ValueError("Could not find 'test' key in the partition file.")

    test_dataset = ActivationDataset(activation_dir=cli_args.activation_dir, file_list=test_files, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for individual attacks

    # --- Attack Loop ---
    reconstruction_losses = {task: [] for task in TASKS}
    criterion = nn.MSELoss()
    
    original_samples = {task: [] for task in TASKS}
    reconstructed_samples = {task: [] for task in TASKS}

    for i, (rf_x, _, cfo_x, _, channel_x, _, z_true, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Attacking Samples"):
        z_true = z_true.squeeze(1).to(device)
        
        # Ground truth inputs for final loss calculation
        true_inputs_cpu = {
            'rf_fingerprinting': rf_x.cpu(),
            'cfo_estimation': cfo_x.cpu(),
            'channel_estimation': channel_x.cpu()
        }
        
        # --- Reconstruction Setup ---
        # Initialize inputs as random tensors that require gradients
        if is_mtl:
            # For MTL, optimize all inputs
            x_0_rf = torch.randn_like(rf_x, requires_grad=True, device=device)
            x_0_cfo = torch.randn_like(cfo_x, requires_grad=True, device=device)
            x_0_channel = torch.randn_like(channel_x, requires_grad=True, device=device)
            trainable_inputs = [x_0_rf, x_0_cfo, x_0_channel]
        else:
            # For single-task, only optimize the relevant input
            task_name = TASKS[0]
            if task_name == 'rf_fingerprinting':
                x_0_rf = torch.randn_like(rf_x, requires_grad=True, device=device)
                trainable_inputs = [x_0_rf]
            elif task_name == 'cfo_estimation':
                x_0_cfo = torch.randn_like(cfo_x, requires_grad=True, device=device)
                trainable_inputs = [x_0_cfo]
            elif task_name == 'channel_estimation':
                x_0_channel = torch.randn_like(channel_x, requires_grad=True, device=device)
                trainable_inputs = [x_0_channel]

        optimizer = optim.Adam(trainable_inputs, lr=cli_args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, min_lr=1e-7)

        # Optimization loop to reconstruct all inputs simultaneously
        t_steps = trange(cli_args.steps, desc="Optimizing all tasks", leave=False)
        for _ in t_steps:
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            if is_mtl:
                # MTL model: use projections dict and sum projections
                projected_tensors = []
                task_data_map = {
                    'rf_fingerprinting': x_0_rf.squeeze(1),
                    'cfo_estimation': x_0_cfo.squeeze(1),
                    'channel_estimation': x_0_channel.squeeze(1)
                }
                
                for task_name in TASKS:
                    inputs = task_data_map[task_name]
                    if task_name == 'rf_fingerprinting':
                        # MTL eval logic for RF averages projections across slices
                        proj = model['projections'][task_name](inputs)
                        projected_tensors.append(proj.mean(dim=0, keepdim=True))
                    else:
                        proj = model['projections'][task_name](inputs)
                        projected_tensors.append(proj)
                
                projected_sum = torch.sum(torch.stack(projected_tensors), dim=0)
                z_hat = model['encoder'](projected_sum)
            else:
                # Single-task model: use projection and encoder directly
                task_name = TASKS[0]  # Single task
                if task_name == 'rf_fingerprinting':
                    inputs = x_0_rf.squeeze(1)
                elif task_name == 'cfo_estimation':
                    inputs = x_0_cfo.squeeze(1)
                elif task_name == 'channel_estimation':
                    inputs = x_0_channel.squeeze(1)
                
                projected = model['projection'](inputs)
                
                # Handle direct CFO case where encoder might be bypassed
                if task_name == 'cfo_estimation' and getattr(train_args, 'direct_cfo', False):
                    z_hat = projected  # Skip encoder for direct CFO
                else:
                    z_hat = model['encoder'](projected)
                
                if task_name == 'rf_fingerprinting':
                    # Average the activations of all slices to get a single vector per file
                    z_hat = z_hat.mean(dim=0, keepdim=True)
            
            # --- Calculate Losses ---
            reconstruction_loss = criterion(z_hat, z_true)
            
            # L2 regularization on inputs (only for variables being optimized)
            l2_reg = 0
            tv_reg = 0
            
            if is_mtl:
                # For MTL, regularize all inputs
                l2_reg = cli_args.regularization * (torch.norm(x_0_rf) + torch.norm(x_0_cfo) + torch.norm(x_0_channel))
                
                # Total Variation (TV) regularization on inputs
                tv_reg_rf = torch.sum(torch.abs(x_0_rf[:, :, :, 1:] - x_0_rf[:, :, :, :-1]))
                tv_reg_cfo = torch.sum(torch.abs(x_0_cfo[:, :, 1:] - x_0_cfo[:, :, :-1]))
                tv_reg_channel = torch.sum(torch.abs(x_0_channel[:, :, 1:] - x_0_channel[:, :, :-1]))
                tv_reg = cli_args.tv_regularization * (tv_reg_rf + tv_reg_cfo + tv_reg_channel)
            else:
                # For single-task, only regularize the relevant input
                task_name = TASKS[0]
                if task_name == 'rf_fingerprinting':
                    l2_reg = cli_args.regularization * torch.norm(x_0_rf)
                    tv_reg = cli_args.tv_regularization * torch.sum(torch.abs(x_0_rf[:, :, :, 1:] - x_0_rf[:, :, :, :-1]))
                elif task_name == 'cfo_estimation':
                    l2_reg = cli_args.regularization * torch.norm(x_0_cfo)
                    tv_reg = cli_args.tv_regularization * torch.sum(torch.abs(x_0_cfo[:, :, 1:] - x_0_cfo[:, :, :-1]))
                elif task_name == 'channel_estimation':
                    l2_reg = cli_args.regularization * torch.norm(x_0_channel)
                    tv_reg = cli_args.tv_regularization * torch.sum(torch.abs(x_0_channel[:, :, 1:] - x_0_channel[:, :, :-1]))

            loss = reconstruction_loss + l2_reg + tv_reg
            loss.backward()
            optimizer.step()
            scheduler.step(reconstruction_loss) # Schedule based on main objective
            t_steps.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # --- Post-Optimization Analysis ---
        reconstructed_data = {}
        if is_mtl:
            # For MTL, we have all reconstructed inputs
            reconstructed_data = {
                'rf_fingerprinting': x_0_rf.detach().cpu(),
                'cfo_estimation': x_0_cfo.detach().cpu(),
                'channel_estimation': x_0_channel.detach().cpu()
            }
        else:
            # For single-task, only the relevant input was optimized
            task_name = TASKS[0]
            if task_name == 'rf_fingerprinting':
                reconstructed_data['rf_fingerprinting'] = x_0_rf.detach().cpu()
            elif task_name == 'cfo_estimation':
                reconstructed_data['cfo_estimation'] = x_0_cfo.detach().cpu()
            elif task_name == 'channel_estimation':
                reconstructed_data['channel_estimation'] = x_0_channel.detach().cpu()

        print(f"\n--- [Sample {i+1}] Post-Attack Analysis ---")
        for task in TASKS:
            reconstructed_flat = reconstructed_data[task].squeeze(0).flatten()
            true_flat = true_inputs_cpu[task].squeeze(0).flatten()

            loss_val = criterion(reconstructed_flat, true_flat).item()
            reconstruction_losses[task].append(loss_val)
            
            correlation, _ = pearsonr(reconstructed_flat.numpy(), true_flat.numpy())
            
            print(f"  Task: {task:<20} | MSE: {loss_val:.4e}, Correlation: {correlation:.4f}")

        # --- DEBUG: Plot after first sample and exit ---
        print("\n--- Generating debug plot for the first sample ---")
        debug_originals = {task: [true_inputs_cpu[task]] for task in TASKS}
        debug_reconstructions = {task: [reconstructed_data[task]] for task in TASKS}
        visualize_reconstruction(debug_originals, debug_reconstructions, TASKS, 'debug_reconstruction_sample_1.png')
        print("--- Debug plot saved. Exiting. ---")
        break # Exit after the first sample

        # Save some samples for visualization
        if i < 5: 
            for task in TASKS:
                original_samples[task].append(true_inputs_cpu[task])
                reconstructed_samples[task].append(reconstructed_data[task])
    
    # --- Report Results ---
    print("\n--- Final Reconstruction Results (MSE) ---")
    for task, losses in reconstruction_losses.items():
        if losses:
            avg_loss = np.mean(losses)
            print(f"  {task:<20}: {avg_loss:.4e}")

    # --- Visualize ---
    if any(original_samples.values()):
        visualize_reconstruction(original_samples, reconstructed_samples, TASKS, 'reconstruction_attack_results.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct data from latent activations via optimization attack.")
    
    # Required paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--activation_dir', type=str, required=True, help="Directory containing pre-computed activation files.")
    parser.add_argument('--partition_file', type=str, required=True, help="Path to the data partition file.")
    
    # Attack parameters
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the reconstruction optimization.")
    parser.add_argument('--steps', type=int, default=6000, help="Number of optimization steps per sample.")
    parser.add_argument('--regularization', type=float, default=1e-6, help="L2 regularization strength on the reconstructed input.")
    parser.add_argument('--tv_regularization', type=float, default=1e-6, help="Total Variation regularization strength.")
    
    # System Configurations
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of GPU to be used.')

    cli_args = parser.parse_args()
    reconstruct_data(cli_args) 