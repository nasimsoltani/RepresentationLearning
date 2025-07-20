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

from rep_lr.models import ComplexSequenceProjector, Encoder, UpsamplingProjector
from dra_1.py_datasets import ActivationDataset

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
    model_dir = os.path.dirname(cli_args.model_path)
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}. Cannot determine model architecture.")
    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))

    # --- Re-create Model Architecture ---
    is_mtl = getattr(train_args, 'mtl', False)
    if not is_mtl:
        raise ValueError("This attack script currently only supports MTL models.")

    # Reconstruct projections and encoder
    projections = torch.nn.ModuleDict()
    TASKS = train_args.task
    TASK_SEQ_LENS = {'rf_fingerprinting': train_args.slice_len, 'cfo_estimation': 160, 'channel_estimation': 160}

    for task in TASKS:
        if task == 'cfo_estimation':
            # CFO estimation uses a parameter-free upsampling projector
            projections[task] = UpsamplingProjector(output_seq_len=train_args.proj_seq_len)
        else:
            projections[task] = ComplexSequenceProjector(
                input_seq_len=TASK_SEQ_LENS[task],
                output_seq_len=train_args.proj_seq_len,
                hidden_dim=train_args.proj_hidden_dim
            )

    encoder = Encoder(
        slice_size=train_args.proj_seq_len,
        output_dim=train_args.d2,
        dropout=train_args.dropout,
        num_blocks=getattr(train_args, 'encoder_num_blocks', 1)
    )
    
    model = torch.nn.ModuleDict({'projections': projections, 'encoder': encoder}).to(device)
    
    # Load trained weights
    print(f"Loading weights from {cli_args.model_path}")
    checkpoint = torch.load(cli_args.model_path, map_location=device)
    model['projections'].load_state_dict(checkpoint['projections_state_dict'], strict=False)
    model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
    model.eval()
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print("Model loaded and frozen successfully.")

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
        
        # --- Simultaneous Reconstruction Setup ---
        # Initialize all inputs as random tensors that require gradients
        x_0_rf = torch.randn_like(rf_x, requires_grad=True, device=device)
        x_0_cfo = torch.randn_like(cfo_x, requires_grad=True, device=device)
        x_0_channel = torch.randn_like(channel_x, requires_grad=True, device=device)

        trainable_inputs = [x_0_rf, x_0_cfo, x_0_channel]
        optimizer = optim.Adam(trainable_inputs, lr=cli_args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, min_lr=1e-7)

        # Optimization loop to reconstruct all inputs simultaneously
        t_steps = trange(cli_args.steps, desc="Optimizing all tasks", leave=False)
        for _ in t_steps:
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # Get projections of the current variable inputs
            proj_rf = model['projections']['rf_fingerprinting'](x_0_rf.squeeze(1)).mean(dim=0, keepdim=True)
            proj_cfo = model['projections']['cfo_estimation'](x_0_cfo.squeeze(1))
            proj_channel = model['projections']['channel_estimation'](x_0_channel.squeeze(1))

            # Combine projections and get the latent representation
            projected_sum = proj_rf + proj_cfo + proj_channel
            z_hat = model['encoder'](projected_sum)
            
            # --- Calculate Losses ---
            reconstruction_loss = criterion(z_hat, z_true)
            
            # L2 regularization on inputs
            l2_reg = cli_args.regularization * (torch.norm(x_0_rf) + torch.norm(x_0_cfo) + torch.norm(x_0_channel))
            
            # Total Variation (TV) regularization on inputs
            tv_reg_rf = torch.sum(torch.abs(x_0_rf[:, :, :, 1:] - x_0_rf[:, :, :, :-1]))
            tv_reg_cfo = torch.sum(torch.abs(x_0_cfo[:, :, 1:] - x_0_cfo[:, :, :-1]))
            tv_reg_channel = torch.sum(torch.abs(x_0_channel[:, :, 1:] - x_0_channel[:, :, :-1]))
            tv_reg = cli_args.tv_regularization * (tv_reg_rf + tv_reg_cfo + tv_reg_channel)

            loss = reconstruction_loss + l2_reg + tv_reg
            loss.backward()
            optimizer.step()
            scheduler.step(reconstruction_loss) # Schedule based on main objective
            t_steps.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # --- Post-Optimization Analysis ---
        reconstructed_data = {
            'rf_fingerprinting': x_0_rf.detach().cpu(),
            'cfo_estimation': x_0_cfo.detach().cpu(),
            'channel_estimation': x_0_channel.detach().cpu()
        }

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